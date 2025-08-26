import React, { createContext, useContext, useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'auto';

interface ThemeContextType {
  theme: Theme;
  isDark: boolean;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: React.ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(() => {
    // Get theme from localStorage or default to 'auto'
    const stored = localStorage.getItem('langgraph-theme') as Theme;
    return stored || 'auto';
  });

  const [isDark, setIsDark] = useState(false);

  // Function to determine if dark mode should be active
  const getIsDark = (currentTheme: Theme): boolean => {
    if (currentTheme === 'dark') {
      return true;
    } else if (currentTheme === 'light') {
      return false;
    } else {
      // Auto mode - use system preference
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
  };

  // Update theme and localStorage
  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    localStorage.setItem('langgraph-theme', newTheme);
  };

  // Toggle between light and dark (not auto)
  const toggleTheme = () => {
    if (theme === 'auto') {
      // If auto, switch to opposite of current appearance
      setTheme(isDark ? 'light' : 'dark');
    } else {
      // If light/dark, toggle to opposite
      setTheme(theme === 'light' ? 'dark' : 'light');
    }
  };

  // Update isDark when theme changes or system preference changes
  useEffect(() => {
    const updateIsDark = () => {
      const newIsDark = getIsDark(theme);
      setIsDark(newIsDark);
      
      // Update HTML class for Tailwind dark mode
      if (newIsDark) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    };

    updateIsDark();

    // Listen for system theme changes when in auto mode
    if (theme === 'auto') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleChange = () => updateIsDark();
      
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [theme]);

  // Update CSS custom properties for theme colors
  useEffect(() => {
    const root = document.documentElement;
    
    if (isDark) {
      // Dark theme colors
      root.style.setProperty('--color-bg-primary', '#111827');
      root.style.setProperty('--color-bg-secondary', '#1f2937');
      root.style.setProperty('--color-text-primary', '#f9fafb');
      root.style.setProperty('--color-text-secondary', '#d1d5db');
      root.style.setProperty('--color-border', '#374151');
    } else {
      // Light theme colors
      root.style.setProperty('--color-bg-primary', '#ffffff');
      root.style.setProperty('--color-bg-secondary', '#f9fafb');
      root.style.setProperty('--color-text-primary', '#111827');
      root.style.setProperty('--color-text-secondary', '#6b7280');
      root.style.setProperty('--color-border', '#e5e7eb');
    }
  }, [isDark]);

  // Set initial theme class on mount
  useEffect(() => {
    const initialIsDark = getIsDark(theme);
    if (initialIsDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const value: ThemeContextType = {
    theme,
    isDark,
    setTheme,
    toggleTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// Theme-aware component wrapper
export function withTheme<P extends object>(Component: React.ComponentType<P>) {
  return function ThemedComponent(props: P) {
    const { isDark } = useTheme();
    return <Component {...props} isDark={isDark} />;
  };
}

// Hook for theme-aware styles
export function useThemeStyles() {
  const { isDark } = useTheme();
  
  return {
    // Background styles
    bgPrimary: isDark ? 'bg-gray-900' : 'bg-white',
    bgSecondary: isDark ? 'bg-gray-800' : 'bg-gray-50',
    bgTertiary: isDark ? 'bg-gray-700' : 'bg-gray-100',
    
    // Text styles
    textPrimary: isDark ? 'text-gray-100' : 'text-gray-900',
    textSecondary: isDark ? 'text-gray-300' : 'text-gray-600',
    textTertiary: isDark ? 'text-gray-400' : 'text-gray-500',
    
    // Border styles
    border: isDark ? 'border-gray-700' : 'border-gray-200',
    borderSecondary: isDark ? 'border-gray-600' : 'border-gray-300',
    
    // Interactive styles
    hover: isDark ? 'hover:bg-gray-700' : 'hover:bg-gray-100',
    focus: 'focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
    
    // Card styles
    card: isDark 
      ? 'bg-gray-800 border-gray-700 shadow-lg' 
      : 'bg-white border-gray-200 shadow-sm',
    
    // Input styles
    input: isDark
      ? 'bg-gray-700 border-gray-600 text-gray-100 focus:border-primary-500'
      : 'bg-white border-gray-300 text-gray-900 focus:border-primary-500',
    
    // Button styles
    buttonPrimary: 'bg-primary-600 hover:bg-primary-700 text-white',
    buttonSecondary: isDark
      ? 'bg-gray-700 hover:bg-gray-600 text-gray-200 border-gray-600'
      : 'bg-gray-200 hover:bg-gray-300 text-gray-900 border-gray-300',
  };
}

// Theme configuration constants
export const THEME_CONFIG = {
  colors: {
    light: {
      primary: '#3b82f6',
      secondary: '#6b7280',
      success: '#22c55e',
      warning: '#f59e0b',
      error: '#ef4444',
      background: '#ffffff',
      surface: '#f9fafb',
      text: '#111827',
    },
    dark: {
      primary: '#60a5fa',
      secondary: '#9ca3af',
      success: '#4ade80',
      warning: '#fbbf24',
      error: '#f87171',
      background: '#111827',
      surface: '#1f2937',
      text: '#f9fafb',
    },
  },
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
  animations: {
    fast: '150ms',
    normal: '300ms',
    slow: '500ms',
  },
} as const;

// Utility function to get theme color
export function getThemeColor(color: keyof typeof THEME_CONFIG.colors.light, isDark: boolean) {
  return isDark ? THEME_CONFIG.colors.dark[color] : THEME_CONFIG.colors.light[color];
}
