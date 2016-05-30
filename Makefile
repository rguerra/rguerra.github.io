#
# Makefile
# rguerra, 2016-01-08 20:19
#

DIR?="."
PORT?=9969

all:
	@echo "Makefile needs your attention"
serve:
	# This needs http-server to be installed
	# npm install http-server -g
	#
	# To run:
	# $ make http-server DIR="dir/path"
	http-server $(DIR) -p $(PORT)

# vim:ft=make
#
