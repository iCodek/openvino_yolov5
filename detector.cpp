#include "detector.h"

#define IMAGE_LEN 640

bool Detector::parseYolov5(const Blob::Ptr &blob, float cof_threshold,
	std::vector<Rect> &o_rect, std::vector<float> &o_rect_cof,
	std::vector<int> &classId)
{
	//�������ͼ���СΪ[1,3,640,640],��ô��������ĳߴ�Ϊ3*80*80*(4+1)*80 | 40*40 | 20*20
	//����blob->getTensorDesc().getDims()��ȡ���ͷ�Ĳ���������w,h,c,batch_size,class_num
	const int net_grid_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);    // 80/40/20
	const int net_grid_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);    // 80/40/20
	const int batch_size = static_cast<int>(blob->getTensorDesc().getDims()[0]);    // batch_size
	const int anchor_num = static_cast<int>(blob->getTensorDesc().getDims()[1]);    // anchor_num,3
	// item_sizeΪ������������COCO���ԣ����Ϊ80
	const int item_size = static_cast<int>(blob->getTensorDesc().getDims()[4]);     // (4+1+class_num)
	cout << "anchor scale: " << net_grid_h << endl;
	//��������ĳ߶ȣ�����get_anchors���������Ӧ��ê��
	std::vector<int> anchors = get_anchors(net_grid_h);
	//��������ڴ����
	InferenceEngine::LockedMemory<const void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap();
	const float *output_blob = blobMapped.as<float *>();

	int net_grid = net_grid_h;
	std::size_t gi = net_grid * item_size;
	std::size_t ggi = net_grid * gi;
	std::size_t anchor_n = anchor_num;
	//���������Ԥ�����벿��
	for (int n = 0; n < anchor_num; ++n)
		for (int i = 0; i < net_grid; ++i)
			for (int j = 0; j < net_grid; ++j) {
				//��ȡ������Ŷ�
				double box_prob = sigmoid(output_blob[n*ggi + i * gi + j * item_size + 4]);

				if (box_prob < cof_threshold) continue;

				double obj_prob = sigmoid(output_blob[n*ggi + i * gi + j * item_size + 5]);
				// ��ȡ���Ŀ���
				//ע��˴����Ϊ���ĵ�����,��Ҫת��Ϊ�ǵ�����
				double x = output_blob[n*ggi + i * gi + j * item_size + 0];
				double y = output_blob[n*ggi + i * gi + j * item_size + 1];
				double w = output_blob[n*ggi + i * gi + j * item_size + 2];
				double h = output_blob[n*ggi + i * gi + j * item_size + 3];
				//��ȡ����������
				double max_prob = 0;
				int idx = 0;
				for (int t = 5; t < item_size; ++t) {
					double tp = sigmoid(output_blob[n*ggi + i * gi + j * item_size + t]);
					if (tp > max_prob) {
						max_prob = tp;
						idx = t - 5;
					}
				}

				float cof = box_prob * max_prob;
				//���ڱ߿����Ŷ�С����ֵ�ı߿�,������������ֵ,�����м�����ټ�����
				if (cof < cof_threshold)  continue;
				//Ԥ������
				x = (sigmoid(x) * 2 - 0.5 + j)* 640.0f / net_grid;
				y = (sigmoid(y) * 2 - 0.5 + i)* 640.0f / net_grid;
				w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
				h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

				double r_x = x - w / 2;
				double r_y = y - h / 2;
				cv::Rect rect = cv::Rect(round(r_x), round(r_y), round(w), round(h));
				//�����������vector��
				o_rect.push_back(rect);
				o_rect_cof.push_back(cof);
				classId.push_back(idx);
			}
	return true;
}

bool Detector::init(std::string xml_path) {
	try {
		_xml_path = xml_path;
		// ������������Core����������Ҫ�ӵ�ǰ·������plugins.xml�ļ�
		InferenceEngine::Core ie;
		//��ѯ֧��Ӳ���豸
		std::vector<std::string> availableDev = ie.GetAvailableDevices();
		for (int i = 0; i < availableDev.size(); i++) {
			cout << "supported device name: " << availableDev[i].c_str() << endl;
		}
		//��IR���ؼ��ģ��
		InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(_xml_path);
		cnnNetwork.setBatchSize(1);
		// ��������ͷ��������
		InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
		InferenceEngine::InputInfo::Ptr input = inputInfo.begin()->second;
		_input_name = inputInfo.begin()->first;//����ͷ������
		input->setPrecision(InferenceEngine::Precision::FP32);
		input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
		InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
		InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
		cnnNetwork.reshape(inputShapes);
		//����������ã�YOLOV5�����������
		_outputinfo = InferenceEngine::OutputsDataMap(cnnNetwork.getOutputsInfo());
		for (auto& output : _outputinfo) {
			cout << output.first << endl;
			output.second->setPrecision(InferenceEngine::Precision::FP32);
		}
		//��ȡ��ִ������
		_network = ie.LoadNetwork(cnnNetwork, "GPU");
		return true;
	}
	catch (std::exception & e) {
		return false;
	}
}

bool Detector::process_frame(Mat &curFrame, std::vector<Object> &detectedObj)
{
	cv::Mat resize_img = letterBox(curFrame);

	std::size_t img_size = 640 * 640;

	//������������
	InferenceEngine::InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
	InferenceEngine::Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
	InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
	float* blob_data = blobMapped.as<float*>();

	//nchw
	//������ͼ�������Blob
	for (std::size_t row = 0; row < IMAGE_LEN; row++) {
		for (std::size_t col = 0; col < IMAGE_LEN; col++) {
			for (std::size_t ch = 0; ch < 3; ch++) {
				//��������ͼ���ÿ�����ر��浽blob_data��Ŀǰ�����blob_data������ֵ
				blob_data[img_size*ch + row * IMAGE_LEN + col] = float(resize_img.at<cv::Vec3b>(row, col)[ch] / 255.0f);
			}
		}
	}
	//ִ������
	infer_request->Infer();

	//��ȡ������,���浽origin_rect,origin_rect_cof��classId
	std::vector<Rect> origin_rect;
	std::vector<float> origin_rect_cof;
	std::vector<int> classId;

	//���ģ����֮ǰ���ռ�ָ��
	std::vector<InferenceEngine::Blob::Ptr> blobs;
	//blobs�������������������뵽parseYolov5�������н���
	for (auto &output : _outputinfo) {
		auto output_name = output.first;
		InferenceEngine::Blob::Ptr blob = infer_request->GetBlob(output_name);
		blobs.push_back(blob);
	}
	//blob�Ĵ�СΪ3
	for (int i = 0; i < blobs.size(); i++) {
		float th = 0.5;
		//TODO:���������Сʹ�ò�ͬ��ֵ�����Լ�����
		if (i == 0) { th = 0.55; }  //СĿ���ϸ�Ҫ��
		if (i == 1) { th = 0.45; }  //��Ŀ��ſ�Ҫ��
		if (i == 2) { th = 0.40; }

		//���ڱ�������������ʱvector 
		std::vector<cv::Rect> origin_rect_temp;
		std::vector<float> origin_rect_cof_temp;
		//����blobs
		parseYolov5(blobs[i], th, origin_rect_temp, origin_rect_cof_temp, classId);
		origin_rect.insert(origin_rect.end(), origin_rect_temp.begin(), origin_rect_temp.end());
		origin_rect_cof.insert(origin_rect_cof.end(), origin_rect_cof_temp.begin(), origin_rect_cof_temp.end());
	}
	//�������������ս��
	std::vector<int> final_id;
	//TODO:�˴�����ֵ��Ҫ����
	//����opencv dnnģ���е�NMSBoxes���зǼ������ƴ���
	cv::dnn::NMSBoxes(origin_rect, origin_rect_cof, _conf_threshold, _nms_area_threshold, final_id);
	//����final_id��ȡ���ս��
	for (int i = 0; i < final_id.size(); ++i) {
		cv::Rect resize_rect = origin_rect[final_id[i]];
		//����detect2origin��������ӳ�䵽ԭͼ
		cv::Rect rawrect = detect2origin(resize_rect, ratio, topPad, leftPad);
		//��������ݽṹObject���浽vector
		cout << "is: " << final_id[i] << endl;
		cout << "is: " << classId[final_id[i]] << endl;
		detectedObj.push_back(Object{
			origin_rect_cof[final_id[i]],
			className[classId[final_id[i]]],rawrect,
			});
	}
	return true;
}

double Detector::sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

const int anchorBig = 640 / 8;            //8���²���
const int anchorMid = 640 / 16;           //16���²���
const int anchorSml = 640 / 32;           //32���²���
 /*yolov5s.yaml����
 anchors:
   - [10,13,16,30,32,23]        # P3/8
   - [30,61,62,45,59,119]       # P4/16
   - [116,90,156,198,373,326]   # P5/32
 */
const int aBig[6] = { 10,13,16,30,32,23 };
const int aMid[6] = { 30,61,62,45,59,119 };
const int aSml[6] = { 116,90,156,198,373,326 };
//��ȡê��
std::vector<int> Detector::get_anchors(int net_grid)
{
	std::vector<int> anchors(6);
	if (net_grid == anchorBig) { anchors.insert(anchors.begin(), aBig, aBig + 6); }
	else if (net_grid == anchorMid) { anchors.insert(anchors.begin(), aMid, aMid + 6); }
	else if (net_grid == anchorSml) { anchors.insert(anchors.begin(), aSml, aSml + 6); }
	return anchors;
}
//ͼ�����������
cv::Mat Detector::letterBox(Mat src)
{
	if (src.empty()) { std::cout << "input image invalid" << endl;  return cv::Mat(); }
	//����Ϊ���߿�ͼ������
	int in_w = src.cols;
	int in_h = src.rows;
	int tar_w = 640;
	int tar_h = 640;
	//�ĸ����ű���Сѡ�ĸ�
	ratio = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
	int inside_w = std::round(in_w * ratio);
	int inside_h = std::round(in_h * ratio);
	int pad_w = tar_w - inside_w;
	int pad_h = tar_h - inside_h;
	//�ڲ�ͼ��resize
	cv::Mat resize_img;
	cv::resize(src, resize_img, cv::Size(inside_w, inside_h));  //��С��Resize
	cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
	pad_w = pad_w / 2;
	pad_h = pad_h / 2;
	//���߿�����ɫ
	topPad = int(std::round(pad_h - 0.1));
	btmPad = int(std::round(pad_h + 0.1));
	leftPad = int(std::round(pad_w - 0.1));
	rightPad = int(std::round(pad_w + 0.1));

	cv::copyMakeBorder(resize_img, resize_img, topPad, btmPad, leftPad, rightPad, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	return resize_img;
}
//��ԭ
//��detect�õ���xywhת���ص�ԭͼxywh
cv::Rect Detector::detect2origin(const Rect &det_rect, float rate_to, int top, int left)
{
	//detect����ת�����ڲ���ͼ����
	int inside_x = det_rect.x - left;
	int inside_y = det_rect.y - top;
	int ox = std::round(float(inside_x) / rate_to);
	int oy = std::round(float(inside_y) / rate_to);
	int ow = std::round(float(det_rect.width) / rate_to);
	int oh = std::round(float(det_rect.height) / rate_to);

	cv::Rect origin_rect(ox, oy, ow, oh);
	return origin_rect;
}
