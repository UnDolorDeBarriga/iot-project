# Makefile for infer_emnist
CC      := gcc
CFLAGS  := -O3 -Wall -Wextra -std=c11
INCLUDE := /usr/local/include
LIBDIR  := /usr/local/lib

TARGET := infer_emnist
SRCS   := main.c
OBJS   := $(SRCS:.c=.o)

LDLIBS := -L$(LIBDIR) -ltensorflowlite_c -lpthread -lm -ldl

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@"
	$(CC) $(CFLAGS) -I$(INCLUDE) $^ $(LDLIBS) -o $@

%.o: %.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
