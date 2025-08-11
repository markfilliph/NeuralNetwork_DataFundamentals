/**
 * Main navigation component with authentication and responsive design
 */

'use client';

import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Avatar,
  Menu,
  MenuItem,
  Chip,
} from '@mui/material';
import {
  MenuOutlined,
  AccountCircleOutlined,
  LogoutOutlined,
  DashboardOutlined,
  DatasetOutlined,
  ModelTrainingOutlined,
} from '@mui/icons-material';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';

export default function Navigation() {
  const router = useRouter();
  const { user, isAuthenticated, logout } = useAuth();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleMenuClose();
    router.push('/login');
  };

  if (!isAuthenticated) {
    return (
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🚀 DAPP - Data Analysis Platform
          </Typography>
          <Button color="inherit" onClick={() => router.push('/login')}>
            Login
          </Button>
        </Toolbar>
      </AppBar>
    );
  }

  return (
    <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Toolbar>
        {/* Logo and Title */}
        <Typography
          variant="h6"
          component="div"
          sx={{ cursor: 'pointer' }}
          onClick={() => router.push('/')}
        >
          🚀 DAPP
        </Typography>

        {/* Navigation Links */}
        <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' }, ml: 4 }}>
          <Button
            color="inherit"
            startIcon={<DashboardOutlined />}
            onClick={() => router.push('/')}
          >
            Dashboard
          </Button>
          <Button
            color="inherit"
            startIcon={<DatasetOutlined />}
            onClick={() => router.push('/datasets')}
          >
            Datasets
          </Button>
          <Button
            color="inherit"
            startIcon={<ModelTrainingOutlined />}
            onClick={() => router.push('/models')}
          >
            Models
          </Button>
        </Box>

        {/* User Info and Menu */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip
            label={user?.role || 'User'}
            size="small"
            color="secondary"
            variant="outlined"
          />
          
          <IconButton
            color="inherit"
            onClick={handleMenuOpen}
            sx={{ ml: 1 }}
          >
            <Avatar sx={{ width: 32, height: 32 }}>
              {user?.email?.charAt(0).toUpperCase()}
            </Avatar>
          </IconButton>
          
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
          >
            <MenuItem onClick={handleMenuClose}>
              <AccountCircleOutlined sx={{ mr: 1 }} />
              Profile
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <LogoutOutlined sx={{ mr: 1 }} />
              Logout
            </MenuItem>
          </Menu>
        </Box>

        {/* Mobile Menu Button */}
        <IconButton
          color="inherit"
          sx={{ display: { md: 'none' }, ml: 1 }}
          onClick={() => {
            // Handle mobile menu
          }}
        >
          <MenuOutlined />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
}