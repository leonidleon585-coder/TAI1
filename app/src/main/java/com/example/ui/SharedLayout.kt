package com.example.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Common Layout Constants for consistent 8dp-grid padding.
 */
object LayoutConstants {
    val PaddingSmall = 8.dp
    val PaddingNormal = 16.dp
    val PaddingLarge = 24.dp
    
    val ShapeCornerNormal = 12.dp
    val BorderWidthThin = 1.dp
    
    val ColorCyberGreen = Color(0xFF10B981)
    val ColorCyberBlue = Color(0xFF3B82F6)
    val ColorCyberPurple = Color(0xFF8B5CF6)
    val ColorDarkBg = Color(0xFF030712)
    val ColorDarkSurface = Color(0xFF0F172A)
    val ColorBorder = Color(0xFF1E293B)
}

/**
 * Standard Neon-Border Card used across all screens.
 */
@Composable
fun CyberCard(
    modifier: Modifier = Modifier,
    borderColor: Color = LayoutConstants.ColorBorder,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .border(
                width = LayoutConstants.BorderWidthThin,
                color = borderColor,
                shape = RoundedCornerShape(LayoutConstants.ShapeCornerNormal)
            ),
        shape = RoundedCornerShape(LayoutConstants.ShapeCornerNormal),
        colors = CardDefaults.cardColors(
            containerColor = LayoutConstants.ColorDarkSurface
        ),
        content = content
    )
}

/**
 * Standard Header for screen sections.
 */
@Composable
fun ScreenHeader(
    title: String,
    subtitle: String,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(bottom = LayoutConstants.PaddingNormal)
    ) {
        Text(
            text = title.uppercase(),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = LayoutConstants.ColorCyberGreen,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.5.sp
        )
        Text(
            text = subtitle,
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray,
            fontFamily = FontFamily.Monospace
        )
    }
}

/**
 * Renders a high-fidelity visual graph showing the live count of active neurons
 * during generation/training steps.
 */
@Composable
fun SynapticActivityGraph(
    history: List<Int>,
    maxNeurons: Int = 64,
    modifier: Modifier = Modifier
) {
    CyberCard(modifier = modifier) {
        Column(modifier = Modifier.padding(LayoutConstants.PaddingNormal)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "INTERNAL SYNAPTIC ACTIVITY",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    fontFamily = FontFamily.Monospace
                )
                
                val currentActive = history.lastOrNull() ?: 0
                Text(
                    text = "$currentActive / $maxNeurons active",
                    style = MaterialTheme.typography.labelSmall,
                    color = LayoutConstants.ColorCyberGreen,
                    fontFamily = FontFamily.Monospace
                )
            }
            
            Spacer(modifier = Modifier.height(LayoutConstants.PaddingSmall))
            
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(100.dp)
                    .background(Color.Black.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                    .border(LayoutConstants.BorderWidthThin, Color(0xFF334155), RoundedCornerShape(8.dp))
                    .padding(8.dp)
            ) {
                if (history.isEmpty()) {
                    Text(
                        text = "Awaiting synaptic activation...",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.DarkGray,
                        modifier = Modifier.align(Alignment.Center),
                        fontFamily = FontFamily.Monospace
                    )
                } else {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val width = size.width
                        val height = size.height
                        val pointsCount = history.size
                        
                        val xSpacing = if (pointsCount > 1) width / (pointsCount - 1) else width
                        val yRatio = height / maxNeurons.toFloat()
                        
                        val path = Path()
                        val fillPath = Path()
                        
                        history.forEachIndexed { i, activeCount ->
                            val x = i * xSpacing
                            val y = height - (activeCount * yRatio).coerceIn(0f, height)
                            
                            if (i == 0) {
                                path.moveTo(x, y)
                                fillPath.moveTo(x, height)
                                fillPath.lineTo(x, y)
                            } else {
                                path.lineTo(x, y)
                                fillPath.lineTo(x, y)
                            }
                            
                            if (i == pointsCount - 1) {
                                fillPath.lineTo(x, height)
                                fillPath.close()
                            }
                        }
                        
                        // Draw filled background gradient
                        drawPath(
                            path = fillPath,
                            brush = Brush.verticalGradient(
                                colors = listOf(
                                    LayoutConstants.ColorCyberGreen.copy(alpha = 0.25f),
                                    Color.Transparent
                                )
                            )
                        )
                        
                        // Draw neon stroke line
                        drawPath(
                            path = path,
                            color = LayoutConstants.ColorCyberGreen,
                            style = Stroke(width = 2.dp.toPx())
                        )
                        
                        // Draw active indicator dots on nodes
                        history.forEachIndexed { i, activeCount ->
                            val x = i * xSpacing
                            val y = height - (activeCount * yRatio).coerceIn(0f, height)
                            drawCircle(
                                color = LayoutConstants.ColorCyberBlue,
                                radius = 2.5.dp.toPx(),
                                center = Offset(x, y)
                            )
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(LayoutConstants.PaddingSmall))
            
            Text(
                text = "Each peak represents live hidden-layer ReLU states firing in memory during generation step forward passes.",
                style = MaterialTheme.typography.labelSmall,
                color = Color.Gray,
                fontFamily = FontFamily.Monospace,
                lineHeight = 12.sp
            )
        }
    }
}
