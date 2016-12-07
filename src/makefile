PROGRAM="lstm_classification.py"
REPORT_PATH="tmp"
REPORT_TARGET="summaries-group-0-iter-1000"

run: 
	python3 $(PROGRAM)

report:
	tensorboard --logdir=${REPORT_PATH}/${REPORT_TARGET}

clean:
	rm -rf $(REPORT_PATH)
	rm -rf __*__

