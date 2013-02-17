CC              := g++
CFLAGS          := -I/usr/local/include/opencv  -L/usr/local/lib
OBJECTS         := 
LIBRARIES       := -lopencv_nonfree -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d

.PHONY: all clean

all: test

test: 
	$(CC) $(CFLAGS) -o test key_points_test.cpp $(LIBRARIES)
        
clean:
	rm -f *.o
