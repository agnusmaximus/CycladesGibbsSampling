FLAGS=-Ofast -std=c++11 -omp

ising:
	rm -f ising_bin
	g++-5 $(FLAGS) src/GibbsSamplingIsing.cpp -o ising_bin
	./ising_bin
