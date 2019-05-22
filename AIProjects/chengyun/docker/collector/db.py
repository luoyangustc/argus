import datetime
import json
import logging
import os

from flask import Flask, jsonify, request
from schema import Or, Schema, SchemaError

import cx_Oracle

os.environ["NLS_LANG"] = ".AL32UTF8"

logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")


def fetch_from_db(start_time: datetime.datetime, end_time: datetime.datetime, dmdm: str, table):
    connection = cx_Oracle.Connection("CYZX", "CYZX", "10.118.84.24/PDCP")
    connection.ping()
    logging.debug("connect db ok")
    cursor = connection.cursor()
    cursor.arraysize = 100
    # TODO: add limit?
    cursor.execute("""
        SELECT JGSJ,DMDM,HPHM,HPYS,CLSD,CDBH,TPID,CSYS FROM {} T WHERE T.HPYS = '01' AND T.JGSJ >= :start_time AND T.JGSJ < :end_time AND (T.DMDM LIKE :dmdm ) """.format(table),
                   start_time=start_time, end_time=end_time, dmdm=dmdm)

    r = []
    for row in cursor:
        row_json = {"JGSJ": row[0].timestamp(), "JGSJ_STR": row[0].isoformat(), "DMDM": row[1], "HPHM": row[2],
                    "HPYS": row[3], "CLSD": row[4], "CDBH": row[5], "TPID": row[6], "CSYS": row[7]}
        # logging.debug("fetch row %s", json.dumps(row_json, ensure_ascii=False))
        r.append(row_json)
    logging.info("fetch over, row len %s", len(r))
    connection.close()
    return r


def format_file_name(line):
    return datetime.datetime.fromtimestamp(line["JGSJ"]).strftime("%Y/%m/%d/%H") + "/{TPID}_{HPHM}_{HPYS}_{CSYS}_hptp.jpg".format(**line)


def process_data_from_db(r, param):
    for line in r:
        line['file_name'] = format_file_name(line)
    base_url = "/static/"+param["camera_ip"]+"/"
    return {"imgs": r, "param": param, "base_url": base_url}


app = Flask(__name__, static_folder="/mnt/samba", static_url_path="/static")
app.config["JSON_AS_ASCII"] = False

fetch_imgs_schema = Schema({"camera_id": str, "camera_ip": str,
                            "start_time": Or(float, int), "duration": int})


def fetch_imgs_raw(params):
    fetch_imgs_schema.validate(params)
    start_time = datetime.datetime.fromtimestamp(params["start_time"])
    end_time = start_time + datetime.timedelta(seconds=params["duration"])
    logging.info("req %s %s %s", params, start_time, end_time)
    r1 = fetch_from_db(start_time=start_time,
                       end_time=end_time, dmdm=params["camera_id"], table="PDDB.SS_CPNEW")
    r2 = fetch_from_db(start_time=start_time,
                       end_time=end_time, dmdm=params["camera_id"], table="PDDB.LS_CPNEW")
    r = process_data_from_db(r1+r2, params)
    return r


@app.errorhandler(500)
def page_not_found(e):
    return jsonify(code=500, error=str(e)), 500


@app.errorhandler(404)
def page_not_found(e):
    return jsonify(code=404, error=str(e)), 404


@app.errorhandler(SchemaError)
def page_not_found(e):
    return jsonify(code=400, error=str(e)), 400


@app.route("/v1/fetch_imgs", methods=["POST"])
def fetch_imgs():
    return jsonify(fetch_imgs_raw(request.json))


if __name__ == "__main__":
    app.run(debug=False, port=7756, host="0.0.0.0", threaded=True)
