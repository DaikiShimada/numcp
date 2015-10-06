# src directory
EXAMPLE_DIR := example

.PHONY: all $(EXAMPLE_DIR)

all: $(EXAMPLE_DIR)

$(EXAMPLE_DIR):
	$(MAKE) -C $@
