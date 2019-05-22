if [ "$QBOXROOT" = "" ]; then
	QBOXROOT=$(cd ../; pwd)
	export QBOXROOT
fi
if [ "$ATARAXIAROOT" = "" ]; then
    ATARAXIAROOT=$QBOXROOT/ataraxia
    export ATARAXIAROOT
fi