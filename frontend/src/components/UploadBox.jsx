import { useState } from "react";
import * as XLSX from "xlsx";

function UploadBox({ onAnalyze }) {
  const [file, setFile] = useState(null);
  const [limit, setLimit] = useState("");

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload a file");
      return;
    }

    let csvText;

    if (file.name.endsWith(".xlsx")) {
      const data = await file.arrayBuffer();
      const workbook = XLSX.read(data);

      let targetSheet = null;

      for (let name of workbook.SheetNames) {
        const worksheet = workbook.Sheets[name];
        const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

        if (json.length > 0) {
          const headers = json[0].map(h => String(h).toLowerCase());
          const hasAmount = headers.some(h => h.includes("amount"));
          const hasDate = headers.some(h => h.includes("date"));

          if (hasAmount && hasDate) {
            targetSheet = worksheet;
            break;
          }
        }
      }

      if (!targetSheet) {
        alert("Could not find transaction sheet.");
        return;
      }

      csvText = XLSX.utils.sheet_to_csv(targetSheet);

    } else {
      csvText = await file.text();
    }

    onAnalyze(csvText, parseFloat(limit));
  };

  return (
    <div className="card">
      <h2>Upload Paytm File</h2>

      <label className="file-upload">
        <input
          type="file"
          accept=".csv,.xlsx"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <span>
          {file ? file.name : "Choose CSV or Excel file"}
        </span>
      </label>

      <input
        type="number"
        placeholder="Monthly Spending Limit"
        value={limit}
        onChange={(e) => setLimit(e.target.value)}
      />

      <button onClick={handleSubmit}>Analyze</button>
    </div>
  );
}

export default UploadBox;