package com.example

import android.app.Application
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.work.Configuration
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter

class CortexApplication : Application(), Configuration.Provider {

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setMinimumLoggingLevel(Log.INFO)
            .build()

    override fun onCreate() {
        super.onCreate()
        
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                // Check if this is the crash process to avoid recursive crash loop
                val currentProcess = getCurrentProcessNameCompat()
                if (currentProcess.endsWith(":crash_process")) {
                    defaultHandler?.uncaughtException(thread, throwable)
                    return@setDefaultUncaughtExceptionHandler
                }

                // Serialize stacktrace
                val sw = StringWriter()
                val pw = PrintWriter(sw)
                throwable.printStackTrace(pw)
                val stacktrace = sw.toString()

                // Save to crash_log.txt
                try {
                    val logFile = File(filesDir, "crash_log.txt")
                    logFile.writeText(stacktrace)
                } catch (e: Exception) {
                    Log.e("CortexApplication", "Failed to save crash log to file: ${e.message}")
                }

                // Launch CrashActivity
                val intent = Intent(this, CrashActivity::class.java).apply {
                    putExtra("error_stacktrace", stacktrace)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                }
                startActivity(intent)

                // Force kill the current crashed process so the system clean restarts
                android.os.Process.killProcess(android.os.Process.myPid())
                System.exit(10)
            } catch (e: Exception) {
                // If anything crashes during crash handling, fallback to default OS behavior
                defaultHandler?.uncaughtException(thread, throwable)
            }
        }
    }

    private fun getCurrentProcessNameCompat(): String {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            getProcessName()
        } else {
            // Fallback for older APIs
            try {
                val method = Class.forName("android.app.ActivityThread")
                    .getDeclaredMethod("currentProcessName")
                method.isAccessible = true
                method.invoke(null) as String
            } catch (e: Exception) {
                packageName
            }
        }
    }
}
