CC = g++
NAME = ex
LDFLAGS = -lpthread

################################################################################

# Target rules
all: build

build: $(NAME)

$(NAME).o: $(NAME).cpp
	$(CC) -o $@ -c $<

$(NAME): $(NAME).o
	$(CC) -o $@ $+ $(LDFLAGS)

run: build
	./$(NAME)

clean:
	rm -f $(NAME) $(NAME).o

clobber: clean

