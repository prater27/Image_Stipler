
FLAG = -O3 -m64 -Wall

Matching.o: ../include/Matching/Matching.cpp ../include/Matching/Matching.h ../include/Matching/BinaryHeap.h ../include/Matching/Globals.h ../include/Matching/Graph.h 
	g++ $(FLAG) -c ../include/Matching/Matching.cpp -o ../include/Matching.o

BinaryHeap.o: ../include/Matching/BinaryHeap.h ../include/Matching/BinaryHeap.cpp ../include/Matching/Globals.h
	g++ $(FLAG) -c ../include/Matching/BinaryHeap.cpp -o ../include/BinaryHeap.o

Graph.o: ../include/Matching/Graph.h ../include/Matching/Graph.cpp
	g++ $(FLAG) -c ../include/Matching/Graph.cpp -o ../include/Graph.o
	
pathDrawer.o: pathDrawer.cpp ../include/MST.h ../include/Matching/Matching.h ../include/Matching/Graph.h ../include/Christofides.h
	g++ $(FLAG) -c pathDrawer.cpp -o pathDrawer.o `pkg-config --cflags --libs opencv4` `pkg-config --cflags --libs opencv4` -lX11

pathDrawerExec: ../include/Matching.o ../include/BinaryHeap.o ../include/Graph.o
	g++ $(FLAG) ../include/Matching.o ../include/BinaryHeap.o ../include/Graph.o pathDrawer.o -o pathDrawerExec `pkg-config --cflags --libs opencv4` `pkg-config --cflags --libs opencv4` -lX11

drawStippled_Points.o: drawStippled_Points.cpp ../include/MST.h ../include/Matching/Matching.h ../include/Matching/Graph.h ../include/Christofides.h
	g++ $(FLAG) -c drawStippled_Points.cpp -o drawStippled_Points.o `pkg-config --cflags --libs opencv4` `pkg-config --cflags --libs opencv4` -lX11

drawStippled_PointsExec: ../include/Matching.o ../include/BinaryHeap.o ../include/Graph.o
	g++ $(FLAG) ../include/Matching.o ../include/BinaryHeap.o ../include/Graph.o drawStippled_Points.o -o drawStippled_PointsExec `pkg-config --cflags --libs opencv4` `pkg-config --cflags --libs opencv4` -lX11
