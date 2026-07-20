// App.tsx
// -------
// Why it exists:
//   Root React component. Sets up client-side routing with react-router-dom.
//   Thin — no business logic here, just route definitions.
//
// What it does:
//   - Defines three routes: /, /result/:id, /history
//   - Wraps all routes in a shared Layout component.
//
// Imported by:
//   - src/main.tsx

import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<UploadPage />} />
          <Route path="/result/:predictionId" element={<ResultPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
