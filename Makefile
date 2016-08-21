FLAGS=-Ofast -std=c++11 -fopenmp
CC=clang-omp++

ising:
	rm -f ising_bin
	$(CC) $(FLAGS) src/GibbsSamplingIsing.cpp -o ising_bin
	./ising_bin
