#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>

#include <iostream>

#include "videoCapture.h"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.5
#define CONF_THRESH 0.4
#define BATCH_SIZE 1


using namespace cv;
using namespace std;
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

const char *class_name[2] = {"person",  "car"};
static Logger gLogger;
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);


void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


cv::Mat uyvy2rgb(unsigned char *srcYUV, int width, int height)
{
        cv::Mat rgbimg(height,width,CV_8UC3);
	unsigned char *yuv, *rgb;
	unsigned char u, v, y1, y2;
 
	yuv = srcYUV;
	rgb = rgbimg.data;
 
	
 
	int size = width * height;
 
	for(int i = 0; i < size; i += 2)
	{
		y1 = yuv[2*i + 1];
		y2 = yuv[2*i + 3];
		u = yuv[2*i];
		v = yuv[2*i + 2];
 

		rgb[3*i]     = (unsigned char)(y1 + 1.402*(u - 128));
		rgb[3*i + 1] = (unsigned char)(y1 - 0.344*(u - 128) - 0.714*(v - 128));
		rgb[3*i + 2] = (unsigned char)(y1 + 1.772*(v - 128));
 
		rgb[3*i + 3] = (unsigned char)(y2 + 1.375*(u - 128));
		rgb[3*i + 4] = (unsigned char)(y2 - 0.344*(u - 128) - 0.714*(v - 128));
		rgb[3*i + 5] = (unsigned char)(y2 + 1.772*(v - 128));
        }

	return rgbimg;
}




int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file("yolov5s.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;



    hl::VideoCapture cap;
    unsigned char *ptr_cam_frame;
    int bytes_used;
    cv::Mat img;
    cap.init_cam(4, 1280, 960, V4L2_PIX_FMT_UYVY, IO_METHOD_USERPTR);
    cv::Mat uyvy_frame(960, 1280, CV_8UC2);
    //VideoCapture cap(0);
    //VideoCapture cap("rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=0.sdp?");

    // if not success, exit program
    //if (cap.isOpened() == false)
    //{
        //cout << "Cannot open the video camera" << endl;
        //cin.get(); //wait for any key press
        //return -1;
    //}

    //double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    //double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

    //cout << "Resolution of the video : " << dWidth << " x " << dHeight << endl;

    string window_name = "Yolov5 USB Camera";
    namedWindow(window_name);

    while (true)
    {
        cap.get_cam_frame(&ptr_cam_frame, &bytes_used);
	uyvy_frame.data = ptr_cam_frame;
        img=uyvy2rgb(uyvy_frame.data,1280, 960);
        //cap.release_cam_frame();
        //cv::cvtColor(uyvy_frame, rgb_frame, CV_YUV2RGB_UYVY);
        //cv::cvtColor(rgb_frame, rgb_frame, COLOR_GRAY2BGR);
        //cv::cvtColor(uyvy_frame, rgb_frame, CV_YUV2BGRA_UYVY);
        //cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE);
        //cv::imshow( "Display window",rgb_frame); 

        //Mat img;
        //bool bSuccess = cap.read(img); // read a new img from video

        //Breaking the while loop if the frames cannot be captured
        //if (bSuccess == false)
        //{
            //cout << "Video camera is disconnected" << endl;
            //cin.get(); //Wait for any key press
            //break;
        //}

        //if (rgb_frame.empty()) continue;
        //cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB & resize
        cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB & resize
        int b=0;
        int fcount = 1;
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }
       
        
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);

        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }

        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(pr_img, res[j].bbox);
                cv::rectangle(pr_img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                // cv::putText(pr_img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                cv::putText(pr_img, class_name[(int)res[j].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                std::cout << "class"<< class_name[(int)res[j].class_id]<< "ssssss" <<r.x<<r.y<<std::endl;
            }

            imshow(window_name, pr_img);
            cap.release_cam_frame();
        }

        if (waitKey(10) == 27)
        {
            cout << "Esc key is pressed by user. Stoppig the video" << endl;
            break;
        }
        // usleep(50); //防止cpu占满 单线程无所谓
        
    }




        

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    return 0;
}
