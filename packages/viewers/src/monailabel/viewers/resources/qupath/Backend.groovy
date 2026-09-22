package org.monailabel.qupath

import com.google.gson.GsonBuilder
import com.google.gson.ToNumberPolicy
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.time.Duration

/** Session credentials are read once from a private file and never logged. */
class Backend {
    static final json = new GsonBuilder().setObjectToNumberStrategy(ToNumberPolicy.LONG_OR_DOUBLE).create()
    final Map settings
    final HttpClient http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
        .connectTimeout(Duration.ofSeconds(20)).build()
    Backend() {
        def location = System.getenv('MONAILABEL_VIEWER_SESSION')
        if (!location) throw new IllegalStateException('Open a sample in QuPath from the MONAI Label web page.')
        def file = Path.of(location)
        settings = (Map)json.fromJson(Files.readString(file), Map)
        Files.delete(file)
    }
    Object request(String path, Object body = null, boolean binary = false, Map extraHeaders = [:]) {
        def builder = HttpRequest.newBuilder(URI.create(settings.url + path))
            .timeout(Duration.ofSeconds(180))
            .header('Authorization', 'Bearer ' + settings.token)
        extraHeaders.each { name, value -> builder.header(name, value) }
        if (body != null) {
            byte[] bytes = body instanceof byte[] ? (byte[])body : json.toJson(body).getBytes('UTF-8')
            if (!extraHeaders.containsKey('Content-Type'))
                builder.header('Content-Type', body instanceof byte[] ? 'application/octet-stream' : 'application/json')
            builder.POST(HttpRequest.BodyPublishers.ofByteArray(bytes))
        } else builder.GET()
        def response = http.send(builder.build(), HttpResponse.BodyHandlers.ofByteArray())
        if (response.statusCode() >= 400) {
            def text = new String(response.body(), 'UTF-8')
            try { text = json.fromJson(text, Map).detail.toString() } catch (Exception ignored) {}
            throw new IOException(text)
        }
        return binary ? response.body() : json.fromJson(new String(response.body(), 'UTF-8'), Object)
    }
    Object submitRegions(String assetId, Map metadata, byte[] mask) {
        String boundary = 'monailabel-' + UUID.randomUUID().toString()
        def body = new ByteArrayOutputStream()
        body.write(('--' + boundary + '\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n' +
            json.toJson(metadata) + '\r\n--' + boundary +
            '\r\nContent-Disposition: form-data; name="mask"; filename="mask.bin"\r\nContent-Type: application/octet-stream\r\n\r\n').getBytes('UTF-8'))
        body.write(mask)
        body.write(('\r\n--' + boundary + '--\r\n').getBytes('UTF-8'))
        return request('/api/assets/' + assetId + '/region-submissions', body.toByteArray(), false,
            ['Content-Type': 'multipart/form-data; boundary=' + boundary])
    }
}
