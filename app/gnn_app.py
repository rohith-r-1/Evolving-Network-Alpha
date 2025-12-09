from flask import Flask, jsonify
from gnn_inference import get_backtest_summary, get_equity_curve

app = Flask(__name__)

@app.route("/gnn_backtest_summary", methods=["GET"])
def backtest_summary():
    summary = get_backtest_summary()
    return jsonify(summary)

@app.route("/gnn_equity_curve", methods=["GET"])
def equity_curve():
    df = get_equity_curve()
    # Convert Date to string for JSON
    df_out = df.copy()
    df_out["Date"] = df_out["Date"].astype(str)
    return jsonify(df_out.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
