import * as ort from "onnxruntime-node";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Model load globally
let sessionPromise = ort.InferenceSession.create(path.join(__dirname, "../model.onnx"));

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const session = await sessionPromise;
    const {
      oil_close,
      nasdaq_close,
      sp500_close,
      CPI,
      GDP,
      silver_close,
      usd_chf,
    } = req.body;

    const inputArray = [
      parseFloat(oil_close),
      parseFloat(nasdaq_close),
      parseFloat(sp500_close),
      parseFloat(CPI),
      parseFloat(GDP),
      parseFloat(silver_close),
      parseFloat(usd_chf),
    ];

    const tensor = new ort.Tensor("float32", Float32Array.from(inputArray), [1, inputArray.length]);
    const results = await session.run({ input: tensor });
    const prediction = Object.values(results)[0].data[0];

    res.status(200).json({ prediction });
  } catch (err) {
    console.error("Prediction error:", err);
    res.status(500).json({ error: "Prediction failed" });
  }
}
