main: main.cpp
	mpicxx -O1 main.cpp -o main

test: main
	mpisubmit.bg -n 125 -w 00:05:00 -m SMP main -- 1000 100 100 5 0.000001 30