CC              := g++
CFLAGS          := -I/usr/local/include/opencv  -L/usr/local/lib
OBJECTS         := 
LIBRARIES       := -lSegFault -lopencv_nonfree -lopencv_core -lopencv_flann -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_video

.PHONY: all clean

#all: test

test: 
	$(CC) $(CFLAGS) -o track track_points.cpp $(LIBRARIES)
	$(CC) $(CFLAGS) -o test key_points_test.cpp $(LIBRARIES)
	$(CC) $(CFLAGS) -o lk_test lucas-kanade.cpp $(LIBRARIES)
        
clean:
	rm -f *.o
