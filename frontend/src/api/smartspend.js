export async function analyzeSpending(csvText, limit) {
    const response = await fetch("http://127.0.0.1:8000/api/v1/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        raw_data: csvText,
        spending_limit: limit
      })
    });
  
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Something went wrong");
    }
  
    return await response.json();
  }