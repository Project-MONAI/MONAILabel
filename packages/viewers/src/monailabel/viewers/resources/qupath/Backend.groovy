/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package org.monailabel.qupath

import com.google.gson.GsonBuilder
import com.google.gson.ToNumberPolicy
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.time.Duration
import java.security.KeyStore
import java.security.cert.CertificateFactory
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManagerFactory

/** Session credentials are read once from a private file and never logged. */
class Backend {
    static final json = new GsonBuilder().setObjectToNumberStrategy(ToNumberPolicy.LONG_OR_DOUBLE).create()
    final Map settings
    final HttpClient http
    Backend() {
        def location = System.getenv('MONAILABEL_VIEWER_SESSION')
        if (!location) throw new IllegalStateException('Open a sample in QuPath from the MONAI Label web page.')
        def file = Path.of(location)
        settings = (Map)json.fromJson(Files.readString(file), Map)
        Files.delete(file)
        def builder = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(20))
        if (settings.ca_certificate) {
            def certificate = CertificateFactory.getInstance('X.509').generateCertificate(
                new ByteArrayInputStream(settings.ca_certificate.getBytes('UTF-8')))
            def store = KeyStore.getInstance(KeyStore.getDefaultType())
            store.load(null, null)
            store.setCertificateEntry('monailabel', certificate)
            def trust = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
            trust.init(store)
            def context = SSLContext.getInstance('TLS')
            context.init(null, trust.trustManagers, null)
            builder.sslContext(context)
        }
        http = builder.build()
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
