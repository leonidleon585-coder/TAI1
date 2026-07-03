package com.example.ml

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

object WebScraper {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    /**
     * Fetches a webpage and returns sanitized prose text.
     */
    suspend fun fetchAndSanitize(urlStr: String): String = withContext(Dispatchers.IO) {
        val url = if (!urlStr.startsWith("http://") && !urlStr.startsWith("https://")) {
            "https://$urlStr"
        } else {
            urlStr
        }

        val request = Request.Builder()
            .url(url)
            .header("User-Agent", "Mozilla/5.0 (Linux; Android 10; Mobile) AIStudioTrainer/1.0")
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw Exception("HTTP Error: ${response.code} ${response.message}")
        }

        val bodyHtml = response.body?.string() ?: throw Exception("Empty response body")
        return@withContext sanitizeHtml(bodyHtml)
    }

    /**
     * Strips HTML tags, script sections, styling blocks, and extracts pure text.
     */
    private fun sanitizeHtml(html: String): String {
        var clean = html

        // Remove script tags and their contents
        val scriptRegex = "<script[^>]*?>[\\s\\S]*?</script>".toRegex(RegexOption.IGNORE_CASE)
        clean = scriptRegex.replace(clean, "")

        // Remove style tags and their contents
        val styleRegex = "<style[^>]*?>[\\s\\S]*?</style>".toRegex(RegexOption.IGNORE_CASE)
        clean = styleRegex.replace(clean, "")

        // Remove comment blocks
        val commentsRegex = "<!--[\\s\\S]*?-->".toRegex()
        clean = commentsRegex.replace(clean, "")

        // Strip general HTML tags
        val tagsRegex = "<[^>]*>".toRegex()
        clean = tagsRegex.replace(clean, " ")

        // Decode common HTML entities
        clean = clean
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&nbsp;", " ")
            .replace("&copy;", "")
            .replace("&reg;", "")

        // Collapse multiple whitespace/newlines into neat single spaces or simple carriage returns
        val multipleSpacesRegex = "\\s+".toRegex()
        clean = multipleSpacesRegex.replace(clean, " ")

        return clean.trim()
    }
}
