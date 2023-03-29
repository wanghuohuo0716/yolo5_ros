#ifndef VIDEO_CAPTURE_H
#define VIDEO_CAPTURE_H

#include <linux/videodev2.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#define ERR -128

struct buffer {
	void *	start;
	size_t	length;
};

enum io_method {
        IO_METHOD_READ = 1,
        IO_METHOD_MMAP,
        IO_METHOD_USERPTR
};

namespace hl{

class VideoCapture
{
private:
	enum io_method   io;
	int              fd;
	struct buffer    *buffers;
	unsigned int     n_buffers;
	struct v4l2_buffer frame_buf;
	char is_initialised; 
	char is_released;
		
private:
	int set_io_method(enum io_method io_meth);
	int stop_capturing(void);
	int start_capturing(void);
	int uninit_device(void);
	int init_read(unsigned int buffer_size);
	int init_mmap(void);
	int init_userp(unsigned int buffer_size);
	int init_device(unsigned int width, unsigned int height, unsigned int format);
	int close_device(void);
	int open_device(const char *dev_name);
			
public:
    	VideoCapture();
	~VideoCapture();
	
	int init_cam(int index, unsigned int width, unsigned int height, unsigned int format, enum io_method io_meth);
	int deinit_cam();
	int get_cam_frame(unsigned char **pointer_to_cam_data, int *size);
	int release_cam_frame();
};

};

#endif
