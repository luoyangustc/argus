set -exv
# mkdir -p /mnt/samba && mount -t cifs //10.118.84.19/opt /mnt/samba -o username=xx,password=,iocharset=utf8
cd /src
exec gunicorn -w 4 -b 0.0.0.0:7756 -t 60 db:app
