import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import * as ort from "onnxruntime-node";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true })); // ✅ For form submissions
app.use(express.static(path.join(__dirname, "public")));

let session;

// ✅ Load ONNX model
(async () => {
  try {
    const modelPath = path.join(__dirname, "model.onnx");
    session = await ort.InferenceSession.create(modelPath);
    console.log("✅ Model loaded successfully");
    console.log("Model Inputs:", session.inputNames);
    console.log("Model Outputs:", session.outputNames);
  } catch (err) {
    console.error("❌ Model loading failed:", err);
  }
})();

// ✅ Prediction route
app.post("/predict", async (req, res) => {
  try {
    if (!session) {
      return res.status(503).json({ error: "Model not loaded yet, try again later." });
    }

    const {
      oil_close,
      nasdaq_close,
      sp500_close,
      CPI,
      GDP,
      silver_close,
      usd_chf,
    } = req.body;

    console.log("Incoming Data:", req.body);

    const inputArray = [
      parseFloat(oil_close),
      parseFloat(nasdaq_close),
      parseFloat(sp500_close),
      parseFloat(CPI),
      parseFloat(GDP),
      parseFloat(silver_close),
      parseFloat(usd_chf),
    ];

    const inputName = session.inputNames[0]; // ✅ Dynamic input name
    const feeds = {
      [inputName]: new ort.Tensor("float32", Float32Array.from(inputArray), [1, inputArray.length]),
    };

    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    const prediction = results[outputName].data[0];

    console.log("Prediction:", prediction);

    res.json({ prediction });
  } catch (err) {
    console.error("Prediction Error:", err);
    res.status(500).json({ error: "Prediction failed" });
  }
});

// ✅ Local run
if (process.env.NODE_ENV !== "production") {
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
}

// ✅ For Vercel deployment
export default app;
