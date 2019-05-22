
if [ -z $TAG ];then
  VERSION=`date "+%Y%m%d"`
  TAG="ava"
fi

REGISTRY_NAME="reg.qiniu.com"
REGISTRY_USER=""
REGISTRY_PASS=""

REPO_NAME="ataraxialab/atlab-base-"
TAG_NAME=$TAG

docker login -u $REGISTRY_USER -p $REGISTRY_PASS $REGISTRY_NAME

PLATFORM="caffe"

for line in $(ls $PLATFORM/*.Dockerfile); do
  BASE=${line%.*}
  BASE=${BASE//\//-}
  IMAGE_NAME=$REGISTRY_NAME/$REPO_NAME$BASE:$TAG_NAME
  docker build -t $IMAGE_NAME -f $line .
  docker push $IMAGE_NAME
  docker rmi -f $IMAGE_NAME
  echo "----------" build success $IMAGE_NAME "----------"
done;



PLATFORM="caffe2"

for line in $(ls $PLATFORM/*.Dockerfile); do
  BASE=${line%.*}
  BASE=${BASE//\//-}
  IMAGE_NAME=$REGISTRY_NAME/$REPO_NAME$BASE:$TAG_NAME
  docker build -t $IMAGE_NAME -f $line .
  docker push $IMAGE_NAME
  docker rmi -f $IMAGE_NAME
  echo "----------" build success $IMAGE_NAME "----------"
done;


PLATFORM="mxnet"

for line in $(ls $PLATFORM/*.Dockerfile); do
  BASE=${line%.*}
  BASE=${BASE//\//-}
  IMAGE_NAME=$REGISTRY_NAME/$REPO_NAME$BASE:$TAG_NAME
  docker build -t $IMAGE_NAME -f $line .
  docker push $IMAGE_NAME
  docker rmi -f $IMAGE_NAME
  echo "----------" build success $IMAGE_NAME "----------"
done;


PLATFORM="tensorflow"

for line in $(ls $PLATFORM/*.Dockerfile); do
  BASE=${line%.*}
  BASE=${BASE//\//-}
  IMAGE_NAME=$REGISTRY_NAME/$REPO_NAME$BASE:$TAG_NAME
  docker build -t $IMAGE_NAME -f $line .
  docker push $IMAGE_NAME
  docker rmi -f $IMAGE_NAME
  echo "----------" build success $IMAGE_NAME "----------"
done;


