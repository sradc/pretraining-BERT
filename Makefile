
writing_ints:
	rm -f writing_ints.py  \
	&& jupyter nbconvert writing_ints.ipynb --to script  \
	&& nohup ipython writing_ints.py &
