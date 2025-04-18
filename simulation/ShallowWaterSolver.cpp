#include "ShallowWaterSolver.h"

ShallowWaterSolver::ShallowWaterSolver(int nx, int nz, double dx, double dt)
     : nx(nx), nz(nz), dx(dx), dt(dt), g(9.81), eps(1e-4 * dx),
       lambda_decay(0.9), lambda_update(0.1), alpha(0.5), beta(2.0), boundaries(4, REFLECTIVE), h_rest(0.0), pml_width(10) {
    h = Eigen::MatrixXd::Zero(nx, nz);
    H = Eigen::MatrixXd::Zero(nx, nz);
    u = Eigen::MatrixXd::Zero(nx + 1, nz);
    w = Eigen::MatrixXd::Zero(nx, nz + 1);

    sigma = Eigen::MatrixXd::Zero(nx, nx);
    gamma = Eigen::MatrixXd::Zero(nx, nx);

    phi = Eigen::MatrixXd::Zero(nx, nz);
    psi = Eigen::MatrixXd::Zero(nx, nz);

    a_ext = Eigen::Vector2d::Zero();
}

double ShallowWaterSolver::h_bar_x(int i, int j) {
     if (i < 0 || i >= nx || j < 0 || j >= nz) return 0.0;

     if (u(i + 1, j) <= 0) {
         return h(std::min(i + 1, nx - 1), j);
     }

     return h(i, j);
}

double ShallowWaterSolver::h_bar_z(int i, int j) {
     if (i < 0 || i >= nx || j < 0 || j >= nz) return 0.0;

     if (w(i, j + 1) <= 0) {
         return h(i, std::min(j + 1, nz - 1));
     }

     return h(i, j);
}

bool ShallowWaterSolver::is_x_axis_pml_region(int i) const {
    return (i < pml_width) || (i >= nx - pml_width);
}

bool ShallowWaterSolver::is_z_axis_pml_region(int j) const {
    return (j < pml_width) || (j >= nz - pml_width);
}

void ShallowWaterSolver::setTerrain(const Eigen::MatrixXd& terrainHeight) {
    if (terrainHeight.rows() != nx || terrainHeight.cols() != nz) {
        std::cerr << "Terrain size must be equal to nx * nz" << std::endl;
        return;
    }

    H = terrainHeight;
}

void ShallowWaterSolver::setWaterHeight(const Eigen::MatrixXd& waterHeight) {
    if (waterHeight.rows() != nx || waterHeight.cols() != nz) {
        std::cerr << "Water height size must be equal to nx * nz" << std::endl;
        return;
    }

    h = waterHeight;
}

void ShallowWaterSolver::setExternalAcceleration(double ax, double az) {
    a_ext << ax, az;
}

void ShallowWaterSolver::setBoundaryTypes(BoundaryType left, BoundaryType right,
    BoundaryType top, BoundaryType bottom) {
    boundaries[0] = left;
    boundaries[1] = right;
    boundaries[2] = top;
    boundaries[3] = bottom;
}

void ShallowWaterSolver::setupDampingRegion(int width) {
    sigma = Eigen::MatrixXd::Zero(nx, nz);
    gamma = Eigen::MatrixXd::Zero(nx, nz);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nz; j++) {
            if (boundaries[0] == ABSORBING && i < width) {
                double d = 1.0 - static_cast<double>(i) / width;
                sigma(i , j) =  d * d;
            }

            if (boundaries[1] == ABSORBING && i >= nx - width) {
                double d = 1.0 - static_cast<double>(nx - 1 - i) / width;
                sigma(i , j) =  d * d;
            }

            if (boundaries[2] == ABSORBING && j < width) {
                double d = 1.0 - static_cast<double>(j) / width;
                gamma(i , j) =  d * d;
            }

            if (boundaries[3] == ABSORBING && j >= nz - width) {
                double d = 1.0 - static_cast<double>(nz - 1 - j) / width;
                gamma(i , j) =  d * d;
            }
        }
    }

    phi = Eigen::MatrixXd::Zero(nx, nz);
    psi = Eigen::MatrixXd::Zero(nx, nz);
}

void ShallowWaterSolver::step() {
    velocityAdvection();
    heightIntegration();
    velocityIntegration();
    applyBoundaryConditions();
    applyStabilityEnhancements();
}

void ShallowWaterSolver::velocityAdvection() {
    Eigen::MatrixXd u_predictor = Eigen::MatrixXd::Zero(u.rows(), u.rows());
    Eigen::MatrixXd w_predictor = Eigen::MatrixXd::Zero(w.rows(), w.rows());
    Eigen::MatrixXd u_mac = u;
    Eigen::MatrixXd w_mac = w;

    auto interpolate = [&](const Eigen::MatrixXd& field, double x, double z, bool isU,
                            double& min_val, double& max_val) {
        double dx = this->dx;
        int nx = this->nx, nz = this->nz;

        double grid_x = isU ? (x / dx - 0.5) : (x / dx);
        double grid_z = isU ? (z / dx) : (z / dx - 0.5);

        int i0 = std::max(0, std::min(nx - 1, static_cast<int>(floor(grid_x))));
        int j0 = std::max(0, std::min(nz - 1, static_cast<int>(floor(grid_z))));
        int i1 = std::min(nx - 1, i0 + 1);
        int j1 = std::min(nz - 1, j0 + 1);

        double fx = grid_x - i0;
        double fz = grid_z - j0;
        fx = std::clamp(fx, 0.0, 1.0);
        fz = std::clamp(fz, 0.0, 1.0);

        double v00, v10, v01, v11;
        if (isU) {
            v00 = field(i0 + 1, j0);
            v10 = field(i1 + 1, j0);
            v01 = field(i0 + 1, j1);
            v11 = field(i1 + 1, j1);
        } else {
            v00 = field(i0, j0 + 1);
            v10 = field(i1, j0 + 1);
            v01 = field(i0, j1 + 1);
            v11 = field(i1, j1 + 1);
        }

        min_val = std::min({v00, v10, v01, v11});
        max_val = std::max({v00, v10, v01, v11});

        return (1 - fx) * (1 - fz) * v00 + fx * (1 - fz) * v10 + (1 - fx) * fz * v01 + fx * fz * v11;
    };

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nz; j++) {
            double x = (i + 0.5) * dx;
            double z = j * dx;
            double vel_u = u(i + 1, j);
            double vel_w = w(i, j);

            double pred_x = x - vel_u * dt;
            double pred_z = z - vel_w * dt;
            double min_u, max_u;
            u_predictor(i + 1, j) = interpolate(u, pred_x, pred_z, true, min_u, max_u);

            double corr_x = x + vel_u * dt;
            double corr_z = z + vel_w * dt;
            double dummy_min, dummy_max;
            double u_c = interpolate(u_predictor, pred_x, pred_z, true, dummy_min, dummy_max);

            double u_mac_val = u_predictor(i + 1, j) + (u(i + 1, j) - u_c) * 0.5;
            u_mac_val = std::clamp(u_mac_val, min_u, max_u);
            u_mac(i + 1, j) = u_mac_val;
        }
    }
}

void ShallowWaterSolver::heightIntegration() {
    Eigen::MatrixXd h_new = h;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nz; j++) {
            // Equation 4
            double flux_x = (h_bar_x(i, j) * u(i + 1, j) - h_bar_x(i, j) * u(i - 1, j)) / dx;
            double flux_z = (h_bar_z(i, j) * w(i, j + 1) - h_bar_z(i, j) * w(i, j - 1)) / dx;
            double dhdt = -(flux_x + flux_z);

            h(i, j) += dhdt * dt;
        }
    }

}

void ShallowWaterSolver::velocityIntegration() {
    // Equation 8
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nz; j++) {
            double eta_right = H(std::min(i + 1, nx - 1), j) + h(std::min(i + 1, nx - 1), j);
            double eta_left = H(i, j) + h(i, j);

            u(i + 1, j) += ((-g / dx) * (eta_right - eta_left) + a_ext(0)) * dt;
        }
    }

    // Equation 9
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nz; j++) {
            double eta_top = H(i, std::min(j + 1, nz - 1)) + h(i, std::min(j + 1, nz - 1));
            double eta_bottom = H(i, j) + h(i, j);

            w(i, j + 1) += ((-g / dx) * (eta_top - eta_bottom) + a_ext(0)) * dt;
        }
    }
}

void ShallowWaterSolver::applyBoundaryConditions() {
    if (boundaries[0] == REFLECTIVE) {
        u.col(0).setZero();
    }

    if (boundaries[1] == REFLECTIVE) {
        u.col(nx).setZero();
    }

    if (boundaries[2] == REFLECTIVE) {
        w.row(0).setZero();
    }

    if (boundaries[3] == REFLECTIVE) {
        w.row(nz).setZero();
    }

    if (std::find(boundaries.begin(), boundaries.end(), ABSORBING) != boundaries.end()) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < nz; j++) {
                if (is_x_axis_pml_region(i)) {
                    // Equation 10 (Apply damping)
                    h(i, j) += (-sigma(i, j) * (h(i, j) - h_rest) + phi(i, j)) * dt;

                    // Equation 11
                    if (i < nx) {
                        u(i + 1, j) += -0.5 * (sigma(std::min(i + 1, nx - 1), j) + sigma(i, j)) * u(i + 1, j) * dt;
                    }

                    double grad_w = (w(i, j + 1) - w(i, j)) / dx;

                    // Equation 12
                    phi(i, j) += -lambda_update * sigma(i, j) * grad_w * dt;

                    // Equation 13
                    phi(i, j) *= lambda_decay;
                }

                if (is_z_axis_pml_region(j)) {
                    // Equation 21
                    h(i, j) += (-gamma(i, j) * (h(i, j) - h_rest) + psi(i, j)) * dt;

                    // Equation 22
                    if (j < nz) {
                        w(i, j + 1) += -0.5 * (gamma(i, std::min(j + 1, nz - 1)) + gamma(i, j)) * w(i, j + 1) * dt;
                    }

                    double grad_u = (u(i + 1, j) - u(i, j)) / dx;

                    // Equation 23
                    psi(i, j) += -lambda_update * gamma(i, j) * grad_u * dt;

                    // Equation 24
                    psi(i, j) *= lambda_decay;
                }
            }
        }
    }
}

void ShallowWaterSolver::applyStabilityEnhancements() {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nz; j++) {
            h(i, j) = std::max(0.0, h(i, j));
        }
    }

    double speed_limit = alpha * dx / dt;

    for (int i = 0; i <= nx; i++) {
        for (int j = 0; j < nz; j++) {
            u(i, j) = std::max(-speed_limit, std::min(speed_limit, u(i, j)));
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j <= nz; j++) {
            w(i, j) = std::max(-speed_limit, std::min(speed_limit, w(i, j)));
        }
    }

    // for (int i = 0; i < nx; i++) {
    //     for (int j = 0; j < nz; j++) {
    //         double h_avgmax = beta * (dx / (g * dt));
    //     }
    // }
}