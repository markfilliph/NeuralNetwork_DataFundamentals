/**
 * Authentication context for managing user state
 * Provides login, logout, and user information across the app
 */

'use client';

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { apiClient, User, AuthResponse } from '@/lib/api-client';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string, role?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!user && !!token;

  // Initialize auth state from localStorage
  useEffect(() => {
    const initAuth = async () => {
      try {
        const savedToken = localStorage.getItem('auth_token');
        if (savedToken) {
          apiClient.setToken(savedToken);
          const userData = await apiClient.getCurrentUser();
          setUser(userData);
          setToken(savedToken);
        }
      } catch (error) {
        console.error('Auth initialization failed:', error);
        // Clear invalid token
        localStorage.removeItem('auth_token');
        apiClient.clearToken();
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string): Promise<void> => {
    setIsLoading(true);
    try {
      const authData: AuthResponse = await apiClient.login(email, password);
      setUser(authData.user_info);
      setToken(authData.access_token);
    } catch (error) {
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (username: string, email: string, password: string, role: string = 'analyst'): Promise<void> => {
    setIsLoading(true);
    try {
      const authData: AuthResponse = await apiClient.register(username, email, password, role);
      setUser(authData.user_info);
      setToken(authData.access_token);
    } catch (error) {
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = (): void => {
    setUser(null);
    setToken(null);
    apiClient.clearToken();
    // Optionally call backend logout endpoint
    apiClient.logout().catch(console.error);
  };

  const value: AuthContextType = {
    user,
    token,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}