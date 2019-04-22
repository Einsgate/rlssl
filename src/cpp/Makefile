CC = g++
NAME = signal-generator

CPPFLAGS ?= -std=c++11
LDFLAGS = -lpthread

################################################################################

# Target rules
all: build

build: $(NAME)

$(NAME).o: $(NAME).cpp
	$(CC) $(CPPFLAGS) -o $@ -c $<

$(NAME): $(NAME).o
	$(CC)  -o $@ $+ $(LDFLAGS)

run: build
	./$(NAME)

clean:
	rm -f $(NAME) $(NAME).o

clobber: clean

