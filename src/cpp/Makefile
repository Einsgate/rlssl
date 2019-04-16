CC = g++

################################################################################

# Target rules
all: build

build: signal-generator

signal-generator.o: signal-generator.cpp
	$(CC) -o $@ -c $<

signal-generator: signal-generator.o
	$(CC) -o $@ $+

run: build
	./signal-generator

clean:
	rm -f signal-generator signal-generator.o

clobber: clean

