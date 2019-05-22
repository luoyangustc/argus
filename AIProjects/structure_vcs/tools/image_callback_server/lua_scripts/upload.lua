local cjson = require "cjson"
local vips = require "vips"

local file_prefix = "/data/images"

function draw_img(img, right_down_x, right_down_y, left_up_x, left_up_y)
    local ink = {255, 0, 0}
    local img = img:draw_rect(ink, left_up_x, left_up_y, right_down_x - left_up_x, right_down_y - left_up_y)
    return img
end

function exit_when_fail(check_express, err_msg, status_code)
    if not check_express then
        ngx.say(err_msg)
        print(err_msg)
        -- TODO, attempt to set status 403 via ngx.exit after sending out the response status 200
        -- ngx.status = status_code
        ngx.exit(status_code)
    end
end

ngx.req.read_body()
local args, err = ngx.req.get_post_args()
if err == "truncated" then
    -- one can choose to ignore or reject the current request here
    return
end

exit_when_fail(args ~= nil, "failed to get post args: " .. (err or ""), ngx.HTTP_BAD_REQUEST)

local channel_id = args["channelId"]
local json_str = args["json"]
exit_when_fail(channel_id ~= nil and channel_id ~= "", "no channelId", ngx.HTTP_BAD_REQUEST)
exit_when_fail(json_str ~= nil and json_str ~= "", "no json", ngx.HTTP_BAD_REQUEST)

local json_data = cjson.decode(json_str)
local image_raw_data = ngx.decode_base64(json_data["picBase64"])
exit_when_fail(image_raw_data ~= nil, "invalid picBase64", ngx.HTTP_BAD_REQUEST)

local tz = json_data["wzbjsj"]
exit_when_fail(tz ~= nil, "no wzbjsj", ngx.HTTP_BAD_REQUEST)
if tz == 0 then
    tz = os.time()
end
local item_type = json_data["spjghxxlx"]
exit_when_fail(tz ~= nil, "no wzbjsj", ngx.HTTP_BAD_REQUEST)

local img = vips.Image.new_from_buffer(image_raw_data)
exit_when_fail(img ~= nil, "image load failed", ngx.HTTP_INTERNAL_SERVER_ERROR)

local base_dir = file_prefix.."/"..channel_id.."/"..item_type
os.execute("mkdir -p " .. base_dir)
img = draw_img(img, json_data["yxjxzb"], json_data["yxjyzb"], json_data["zsjxzb"], json_data["zsjyzb"])
img:write_to_file(base_dir.."/"..tz..".jpg")

-- remove picBase64
json_data["picBase64"] = ""
local file = io.open(base_dir.."/"..tz..".jpg.json.txt", "w")
file:write(cjson.encode(json_data))
file:close()

ngx.say("ok")