DATA_PATH="./stock_data"
TARGET_PATH="../data"
SUFFIX=".csv"
LIKE="like"
DISLIKE="dislike"

download:
	wget http://cs.drexel.edu/~jk993/data/stock_data.zip
	unzip stock_data.zip

sample: clean
	python datasampling.py 10 $(DATA_PATH)/$(LIKE) $(TARGET_PATH)/$(LIKE) $(SUFFIX)
	python datasampling.py 10 $(DATA_PATH)/$(DISLIKE) $(TARGET_PATH)/$(DISLIKE) $(SUFFIX)

origin: clean
	cp -r $(DATA_PATH)/$(LIKE)/* $(TARGET_PATH)/$(LIKE)/
	cp -r $(DATA_PATH)/$(DISLIKE)/* $(TARGET_PATH)/$(DISLIKE)/

mktargetdir:
	mkdir $(TARGET_PATH)/$(LIKE)
	mkdir $(TARGET_PATH)/$(DISLIKE)

clean:
	rm -rf $(TARGET_PATH)/$(LIKE)/*
	rm -rf $(TARGET_PATH)/$(DISLIKE)/*

