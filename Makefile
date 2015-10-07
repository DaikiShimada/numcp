# src directory
DIRS := example

.PHONY: all $(DIRS)

all: $(DIRS)

$(DIRS):
	$(MAKE) -C $@
