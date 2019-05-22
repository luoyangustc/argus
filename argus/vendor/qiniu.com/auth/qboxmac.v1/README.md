# QBox Authorization

```
<Method> <PathWithRawQuery> #这里的 <PathWithRawQuery> = <Path>[?<RawQuery>]
<Header1>: <Value1>
<Header2>: <Value2>
...
Content-Type: <ContentType>
...
Authorization: QBox <AK>:<Sign>
...

<Body>
```

对于上面这样一个请求，我们构造如下这个待签名的 `<Data>`：

```
<PathWithRawQuery>
[<Body>] #这里的 <Body> 只有在 <ContentType> 为 application/x-www-form-urlencoded 时才签进去。
```

有了 `<Data>`，就可以计算对应的 `<Sign>`，如下：

```
<Sign> = urlsafe_base64( hmac_sha1(<SK>, <Data>) )
```

# QBoxAdmin Authorization

```
<Method> <PathWithRawQuery> #这里的 <PathWithRawQuery> = <Path>[?<RawQuery>]
<Header1>: <Value1>
<Header2>: <Value2>
...
Content-Type: <ContentType>
...
Authorization: QBoxAdmin <SuInfo>:<AK>:<Sign>
...

<Body>
```

对于上面这样一个请求，我们构造如下这个待签名的 `<Data>`：

```
<PathWithRawQuery>
Authorization: QBoxAdmin <SuInfo>

[<Body>] #这里的 <Body> 只有在 <ContentType> 为 application/x-www-form-urlencoded 时才签进去。
```

有了 `<Data>`，就可以计算对应的 `<Sign>`，如下：

```
<Sign> = urlsafe_base64( hmac_sha1(<SK>, <Data>) )
```

其中，`SuInfo`表示`[uid]/[appid]`，当请求方无法获得appid时，设置appid为0，表示appName是default对应的那个appid
