package com.example.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF10B981),      // Neon Green
    secondary = Color(0xFF3B82F6),    // Cyber Blue
    tertiary = Color(0xFF8B5CF6),     // Futuristic Purple
    background = Color(0xFF0B0E1B),   // Deep dark navy
    surface = Color(0xFF151A35),      // Semi-transparent deep slate cards
    onPrimary = Color.White,
    onSecondary = Color.White,
    onBackground = Color(0xFFE2E8F0), // Light slate text
    onSurface = Color(0xFFF1F5F9)     // Bright slate text
)

private val LightColorScheme = DarkColorScheme // Always use premium dark theme for sci-fi training feel

@Composable
fun MyApplicationTheme(
  darkTheme: Boolean = true, // Default to dark theme for immersive cyber visual styling
  dynamicColor: Boolean = false, // Set to false to preserve our custom premium cyber palette
  content: @Composable () -> Unit,
) {
  val colorScheme =
    when {
      dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
        val context = LocalContext.current
        if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
      }

      darkTheme -> DarkColorScheme
      else -> LightColorScheme
    }

  MaterialTheme(colorScheme = colorScheme, typography = Typography, content = content)
}
