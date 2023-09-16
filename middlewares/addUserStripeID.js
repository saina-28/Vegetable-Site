const userModel = require("../models/User")

const addUserStripeID = async(req, res, next)=>{
    try{
        const userid = req.cookies.userid
        const user = await userModel.findOne({_id : userid})
        req.stripe_customer = user.stripe_customer
        next()
    }
    catch(error){
        res.status(500).json({error : error.message})
    }
}
module.exports = addUserStripeID 