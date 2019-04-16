CC = g++
NAME = ex

################################################################################

# Target rules
all: build

build: $(NAME)

$(NAME).o: $(NAME).cpp
	$(CC) -o $@ -c $<

$(NAME): $(NAME).o
	$(CC) -o $@ $+

run: build
	./$(NAME)

clean:
	rm -f $(NAME) $(NAME).o

clobber: clean

