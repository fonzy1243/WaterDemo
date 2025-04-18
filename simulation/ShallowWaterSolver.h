#ifndef SHALLOWWATERSOLVER_H
#define SHALLOWWATERSOLVER_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>

/**
 * Employs the Shallow Water Equations (SWE) using a 2D height field representation
 * of the liquid surface.
 **/
class ShallowWaterSolver {
private:
    int nx, nz;
    double dx, dt;
    double g;
    double eps;

    double lambda_decay, lambda_update;
    double alpha, beta;

    const int pml_width;

    enum BoundaryType { REFLECTIVE, ABSORBING };
    std::vector<BoundaryType> boundaries;

    Eigen::MatrixXd h;              // Water height
    Eigen::MatrixXd H;              // Terrain height

    Eigen::MatrixXd u;              // u-component of v (velocity)
    Eigen::MatrixXd w;              // w-component of v (velocity)

    Eigen::Vector2d a_ext;          // External acceleration

    Eigen::MatrixXd sigma;          // Damping coefficient x-axis
    Eigen::MatrixXd gamma;          // Damping coefficient z-axis
    Eigen::MatrixXd phi, psi;            // Damping field
    double h_rest;                  // Rest water depth

    double h_bar_x(int i, int j);   // h evaluated in upwind x-direction (Equation 5)
    double h_bar_z(int i, int j);   // h evaluated in upwind z-direction (Equation 6)

    [[nodiscard]] bool is_x_axis_pml_region(int i) const;
    [[nodiscard]] bool is_z_axis_pml_region(int j) const;

public:
    ShallowWaterSolver(int nx, int nz, double dx, double dt);

    void setGravity(double g) { this->g = g; }
    void setTerrain(const Eigen::MatrixXd& terrainHeight);
    void setWaterHeight(const Eigen::MatrixXd& waterHeight);
    void setExternalAcceleration(double ax, double az);
    void setBoundaryTypes(BoundaryType left, BoundaryType right, BoundaryType top, BoundaryType bottom);
    void setupDampingRegion(int width);

    void step();                        // Perform one simulation step
    void velocityAdvection();
    void heightIntegration();
    void velocityIntegration();
    void applyBoundaryConditions();
    void applyStabilityEnhancements();

    [[nodiscard]] const Eigen::MatrixXd& getWaterHeight() const { return h; }
    [[nodiscard]] const Eigen::MatrixXd& getTerrainHeight() const { return H; }
    [[nodiscard]] const Eigen::MatrixXd& getXVelocity() const { return u; }
    [[nodiscard]] const Eigen::MatrixXd& getZVelocity() const { return w; }

    // Total water surface height (η = H + h)
    [[nodiscard]] Eigen::MatrixXd getWaterSurface() const;
};



#endif //SHALLOWWATERSOLVER_H
