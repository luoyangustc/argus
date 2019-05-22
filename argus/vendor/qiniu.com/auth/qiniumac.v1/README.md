# Qiniu Authorization

```
<Method> <PathWithRawQuery>
<Header1>: <Value1>
<Header2>: <Value2>
...
Host: <Host>
...
Content-Type: <ContentType>
...
Authorization: Qiniu <AK>:<Sign>
...

<Body>
```

对于上面这样一个请求，我们构造如下这个待签名的 `<Data>`：

```
<Method> <PathWithRawQuery>
Host: <Host>
Content-Type: <ContentType>

[<Body>] #这里的 <Body> 只有在 <ContentType> 存在且不为 application/octet-stream 时才签进去。
```

有了 `<Data>`，就可以计算对应的 `<Sign>`，如下：

```
<Sign> = urlsafe_base64( hmac_sha1(<SK>, <Data>) )
```

# QiniuAdmin Authorization

```
<Method> <PathWithRawQuery>
<Header1>: <Value1>
<Header2>: <Value2>
...
Host: <Host>
...
Content-Type: <ContentType>
...
Authorization: QiniuAdmin <SuInfo>:<AK>:<Sign>
...

<Body>
```

对于上面这样一个请求，我们构造如下这个待签名的 `<Data>`：

```
<Method> <PathWithRawQuery>
Host: <Host>
Content-Type: <ContentType>
Authorization: QiniuAdmin <SuInfo>

[<Body>] #这里的 <Body> 只有在 <ContentType> 存在且不为 application/octet-stream 时才签进去。
```

有了 `<Data>`，就可以计算对应的 `<Sign>`，如下：

```
<Sign> = urlsafe_base64( hmac_sha1(<SK>, <Data>) )
```

其中，`SuInfo`表示`[uid]/[appid]`，当请求方无法获得appid时，设置appid为0，表示appName是default对应的那个appid
