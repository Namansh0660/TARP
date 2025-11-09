from flask import Flask, request, render_template, jsonify
import datetime

app = Flask(__name__)

# Store latest sensor data
sensor_data = {
    "distance": 0,
    "ir": 0,
    "pir": 0,
    "alert": "None",
    "last_motion": "No motion yet"
}

@app.route("/")
def dashboard():
    return render_template("dashboard.html", data=sensor_data)


# ===== ESP8266 POSTS sensor data here =====
@app.route("/update", methods=["POST"])
def update_data():
    content = request.get_json(force=True)

    # Update sensor values if present
    if "distance" in content:
        sensor_data["distance"] = content["distance"]

    if "ir" in content:
        sensor_data["ir"] = content["ir"]

    if "pir" in content:
        sensor_data["pir"] = content["pir"]

    if "alert" in content:
        sensor_data["alert"] = content["alert"]

    # Update timestamp only when motion (PIR = 1)
    if sensor_data["pir"] == 1:
        sensor_data["last_motion"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("Updated sensor data:", sensor_data)

    return jsonify({"status": "success"})


# ===== Dashboard uses this endpoint to auto-refresh =====
@app.route("/status")
def status():
    return jsonify(sensor_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
