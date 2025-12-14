//! WebSocket authentication and authorization

use crate::common::{Result, DeepAlphaError};
use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use uuid::Uuid;

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    None,
    ApiKey,
    Jwt,
    Token,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub method: AuthMethod,
    pub jwt_secret: Option<String>,
    pub api_keys: HashMap<String, UserInfo>,
    pub token_expiry: u64, // seconds
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            method: AuthMethod::None,
            jwt_secret: None,
            api_keys: HashMap::new(),
            token_expiry: 3600, // 1 hour
        }
    }
}

/// User information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub user_id: String,
    pub username: String,
    pub permissions: Vec<String>,
    pub rate_limit: u32,
}

/// Authentication result
#[derive(Debug, Clone)]
pub enum AuthResult {
    Success(UserInfo),
    Failure(String),
    Pending,
}

/// JWT Claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String, // Subject (user ID)
    pub username: String,
    pub permissions: Vec<String>,
    pub exp: u64, // Expiration time
    pub iat: u64, // Issued at
    pub jti: String, // JWT ID
}

/// WebSocket authenticator
pub struct Authenticator {
    config: AuthConfig,
    active_tokens: HashMap<String, Claims>, // In-memory token store
}

impl Authenticator {
    /// Create a new authenticator
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config,
            active_tokens: HashMap::new(),
        }
    }

    /// Authenticate a connection
    pub async fn authenticate(&self, credentials: &AuthCredentials) -> AuthResult {
        match self.config.method {
            AuthMethod::None => AuthResult::Success(UserInfo {
                user_id: "anonymous".to_string(),
                username: "anonymous".to_string(),
                permissions: vec!["read".to_string()],
                rate_limit: 100,
            }),
            AuthMethod::ApiKey => self.authenticate_api_key(&credentials.api_key),
            AuthMethod::Jwt => self.authenticate_jwt(&credentials.token),
            AuthMethod::Token => self.authenticate_token(&credentials.token),
        }
    }

    /// Generate JWT token for user
    pub fn generate_token(&self, user_info: &UserInfo) -> Result<String> {
        if self.config.jwt_secret.is_none() {
            return Err(DeepAlphaError::InvalidInput(
                "JWT secret not configured".to_string(),
            ));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let claims = Claims {
            sub: user_info.user_id.clone(),
            username: user_info.username.clone(),
            permissions: user_info.permissions.clone(),
            exp: now + self.config.token_expiry,
            iat: now,
            jti: Uuid::new_v4().to_string(),
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.config.jwt_secret.as_ref().unwrap().as_ref()),
        ).map_err(|e| DeepAlphaError::InvalidInput(e.to_string()))?;

        Ok(token)
    }

    /// Validate JWT token
    fn authenticate_jwt(&self, token: &str) -> AuthResult {
        if let Some(secret) = &self.config.jwt_secret {
            let token_data = decode::<Claims>(
                token,
                &DecodingKey::from_secret(secret.as_ref()),
                &Validation::default(),
            );

            match token_data {
                Ok(data) => {
                    let claims = data.claims;
                    AuthResult::Success(UserInfo {
                        user_id: claims.sub,
                        username: claims.username,
                        permissions: claims.permissions,
                        rate_limit: 1000,
                    })
                }
                Err(e) => AuthResult::Failure(format!("Invalid token: {}", e)),
            }
        } else {
            AuthResult::Failure("JWT authentication not configured".to_string())
        }
    }

    /// Authenticate with API key
    fn authenticate_api_key(&self, api_key: &str) -> AuthResult {
        if let Some(user_info) = self.config.api_keys.get(api_key) {
            AuthResult::Success(user_info.clone())
        } else {
            AuthResult::Failure("Invalid API key".to_string())
        }
    }

    /// Authenticate with token
    fn authenticate_token(&self, token: &str) -> AuthResult {
        if let Some(claims) = self.active_tokens.get(token) {
            // Check if token is expired
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if claims.exp > now {
                AuthResult::Success(UserInfo {
                    user_id: claims.sub.clone(),
                    username: claims.username.clone(),
                    permissions: claims.permissions.clone(),
                    rate_limit: 1000,
                })
            } else {
                AuthResult::Failure("Token expired".to_string())
            }
        } else {
            AuthResult::Failure("Invalid token".to_string())
        }
    }

    /// Check if user has permission
    pub fn has_permission(user_info: &UserInfo, permission: &str) -> bool {
        user_info.permissions.contains(&permission.to_string()) ||
        user_info.permissions.contains(&"*".to_string())
    }

    /// Check rate limit
    pub fn check_rate_limit(user_info: &UserInfo, current_usage: u32) -> bool {
        current_usage < user_info.rate_limit
    }
}

/// Authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCredentials {
    pub auth_method: AuthMethod,
    pub api_key: String,
    pub token: String,
    pub username: String,
    pub password: String,
}

impl Default for AuthCredentials {
    fn default() -> Self {
        Self {
            auth_method: AuthMethod::None,
            api_key: String::new(),
            token: String::new(),
            username: String::new(),
            password: String::new(),
        }
    }
}

/// Permission levels
#[pyclass]
#[derive(Debug, Clone)]
pub enum Permission {
    #[pyo3(name = "read")]
    Read,
    #[pyo3(name = "write")]
    Write,
    #[pyo3(name = "trade")]
    Trade,
    #[pyo3(name = "admin")]
    Admin,
    #[pyo3(name = "*")]
    All,
}

impl Permission {
    pub fn as_str(&self) -> &'static str {
        match self {
            Permission::Read => "read",
            Permission::Write => "write",
            Permission::Trade => "trade",
            Permission::Admin => "admin",
            Permission::All => "*",
        }
    }
}

/// Rate limiter for connections
pub struct RateLimiter {
    limits: HashMap<String, (u32, u64)>, // (usage_count, last_reset_time)
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
        }
    }

    /// Check if client is within rate limit
    pub fn check_limit(&mut self, client_id: &str, limit: u32) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = self.limits.entry(client_id.to_string()).or_insert((0, now));

        // Reset counter every minute
        if now - entry.1 > 60 {
            entry.0 = 0;
            entry.1 = now;
        }

        if entry.0 < limit {
            entry.0 += 1;
            true
        } else {
            false
        }
    }

    /// Get current usage for client
    pub fn get_usage(&self, client_id: &str) -> Option<u32> {
        self.limits.get(client_id).map(|(usage, _)| *usage)
    }

    /// Reset limit for client
    pub fn reset_limit(&mut self, client_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.limits.insert(client_id.to_string(), (0, now));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_config_default() {
        let config = AuthConfig::default();
        matches!(config.method, AuthMethod::None);
        assert_eq!(config.token_expiry, 3600);
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new();
        let client_id = "test_client";

        // Should allow initial requests
        assert!(limiter.check_limit(client_id, 5));
        assert_eq!(limiter.get_usage(client_id), Some(1));

        // Should reach limit
        for _ in 1..5 {
            assert!(limiter.check_limit(client_id, 5));
        }
        assert!(!limiter.check_limit(client_id, 5));

        // Reset should work
        limiter.reset_limit(client_id);
        assert!(limiter.check_limit(client_id, 5));
        assert_eq!(limiter.get_usage(client_id), Some(1));
    }

    #[test]
    fn test_permissions() {
        let user = UserInfo {
            user_id: "test".to_string(),
            username: "test".to_string(),
            permissions: vec!["read".to_string(), "write".to_string()],
            rate_limit: 100,
        };

        let auth = Authenticator::new(AuthConfig::default());

        assert!(auth.has_permission(&user, "read"));
        assert!(auth.has_permission(&user, "write"));
        assert!(!auth.has_permission(&user, "admin"));
    }
}