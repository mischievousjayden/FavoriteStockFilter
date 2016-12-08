PROGRAM="lstm_classification.py"
REPORT_PATH="tmp"
REPORT_TARGET0="summaries-group-0-iter-1000"
REPORT_TARGET1="summaries-group-1-iter-1000"
REPORT_TARGET2="summaries-group-2-iter-1000"
REPORT_TARGET3="summaries-group-3-iter-1000"

run: 
	python3 $(PROGRAM)

report:
	tensorboard --logdir=${REPORT_PATH}/${REPORT_TARGET0} &
	tensorboard --logdir=${REPORT_PATH}/${REPORT_TARGET1} --port=6007 &
	tensorboard --logdir=${REPORT_PATH}/${REPORT_TARGET2} --port=6008 &
	tensorboard --logdir=${REPORT_PATH}/${REPORT_TARGET3} --port=6009 &

clean:
	rm -rf $(REPORT_PATH)
	rm -rf __*__
	ps -ef | grep tensorboard | awk '{print $$2}' | xargs sudo kill -9

