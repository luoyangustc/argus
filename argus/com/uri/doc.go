/*
Package uri 通用uri资源访问SDK。

issue： https://jira.qiniu.io/browse/ATLAB-215

支持的资源类型：
	网络资源：http://host/xx OR https://host/xx
	七牛Kodo资源：qiniu://uid@zone/bucket/key
	本地资源：file:///xx
	sts资源：sts://ip:port/xxx(暂不支持）

详细用法请参照 example_test.go

使用注意：
	建议设置超时 context.WithTimeout(ctx, 10*time.Second)
	注意 defer resp.Body.Close()

关于rsHost 和 ioHost的设置可以参照  https://cf.qiniu.io/pages/viewpage.action?pageId=16092953&focusedCommentId=16746280#comment-16746280

*/
package uri
