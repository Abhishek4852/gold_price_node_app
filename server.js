// server.js
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
app.use(express.urlencoded({ extended: true })); // for form submission
app.use(express.static("public"));

let session;

// 🔹 Load ONNX model
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

// 🔹 Serve frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// 🔹 Prediction route (for HTML form)
app.post("/predict", async (req, res) => {
  try {
    if (!session) return res.send("Model is still loading, try again later.");

    // Extract form data (as strings)
    const {
      oil_close,
      nasdaq_close,
      sp500_close,
      CPI,
      GDP,
      silver_close,
      usd_chf,
    } = req.body;

    // Convert to numbers
    const inputArray = [
      parseFloat(oil_close),
      parseFloat(nasdaq_close),
      parseFloat(sp500_close),
      parseFloat(CPI),
      parseFloat(GDP),
      parseFloat(silver_close),
      parseFloat(usd_chf),
    ];

    // Create tensor for ONNX
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const tensor = new ort.Tensor("float32", Float32Array.from(inputArray), [1, inputArray.length]);

    const results = await session.run({ [inputName]: tensor });
    const prediction = results[outputName].data[0];

    // Load HTML and show result dynamically
    import("fs").then((fs) => {
      const filePath = path.join(__dirname, "public", "index.html");
      fs.readFile(filePath, "utf8", (err, data) => {
        if (err) return res.send("Error loading HTML file");
        const updatedHtml = data.replace("{{ prediction_text }}", `Predicted Gold Price: ₹${prediction.toFixed(2)}`);
        res.send(updatedHtml);
      });
    });
  } catch (err) {
    console.error("Prediction Error:", err);
    res.status(500).send("Prediction failed.");
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
