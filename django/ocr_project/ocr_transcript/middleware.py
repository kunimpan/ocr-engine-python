# middleware.py
class ContentSecurityPolicyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        # เพิ่ม header CSP ให้กับทุก response
        response["Content-Security-Policy"] = "default-src 'self' data:;"
        return response
