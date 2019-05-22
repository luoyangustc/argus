package conf

var CLIENT_ID = "abcd0c7edcdf914228ed8aa7c6cee2f2bc6155e2"
var CLIENT_SECRET = "fc9ef8b171a74e197b17f85ba23799860ddf3b9c"

var REDIRECT_URI = "<RedirectURL>"
var AUTHORIZATION_ENDPOINT = "<AuthURL>"
var TOKEN_ENDPOINT = "https://acc.qbox.me/oauth2/token"

var FS_HOST = "https://fs.qbox.me"
var EU_HOST = "http://eu.qbox.me"
var MQ_HOST = "http://mq.qbox.me"
var CDNMGR_HOST = "http://cdnmgr.qbox.me:15001"
var STATUS_HOST = "http://api.qiniu.com/status"
var PFOP_HOST = "http://api.qiniu.com/pfop"

var BLOCK_BITS uint = 22
var PUT_CHUNK_SIZE = 256 * 1024 // 256k
var PUT_RETRY_TIMES = 2
var RS_PUT = "/rs-put/"
