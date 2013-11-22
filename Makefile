default:
	echo "Select an option from the file to run"

get_webgraph:
	wget http://www3.nd.edu/~networks/resources/www/www.dat.gz
	gunzip www.dat.gz

DEST=amor:.www/cddemo/
copy_files:
	scp www-normalized.dat $(DEST)
	scp webgraph-S.p* $(DEST)


