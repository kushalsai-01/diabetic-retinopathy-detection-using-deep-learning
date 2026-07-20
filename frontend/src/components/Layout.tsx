import { Outlet, NavLink } from "react-router-dom";

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <nav className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-teal-500 flex items-center justify-center text-gray-950 font-bold text-lg shadow-lg shadow-teal-500/20">
                DR
              </div>
              <span className="font-semibold text-lg tracking-tight text-teal-400">
                RetinaGuard AI
              </span>
            </div>
            <div className="flex gap-4">
              <NavLink
                to="/"
                className={({ isActive }) =>
                  `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-teal-500/10 text-teal-400 border border-teal-500/20"
                      : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                  }`
                }
              >
                Predict
              </NavLink>
              <NavLink
                to="/history"
                className={({ isActive }) =>
                  `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-teal-500/10 text-teal-400 border border-teal-500/20"
                      : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                  }`
                }
              >
                History
              </NavLink>
            </div>
          </div>
        </div>
      </nav>
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
      <footer className="bg-gray-950 border-t border-gray-900 py-6 text-center text-sm text-gray-600">
        &copy; {new Date().getFullYear()} RetinaGuard AI. Assisted Clinical Screening System.
      </footer>
    </div>
  );
}
